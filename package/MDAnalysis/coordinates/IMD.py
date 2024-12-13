"""
MDAnalysis IMDReader
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: IMDReader
   :members:
   :inherited-members:

"""

from MDAnalysis.coordinates import core
from MDAnalysis.lib.util import store_init_arguments

from MDAnalysis.coordinates.base import (
    ReaderBase,
    FrameIteratorBase,
    FrameIteratorAll,
)

import numbers
import warnings

# NOTE: changeme
from imdclient import IMDClient
import logging

logger = logging.getLogger("imdclient.IMDClient")

# NOTE: think of other edge cases as well- should be robust
def parse_host_port(filename):
    if not filename.startswith("imd://"):
        raise ValueError("IMDReader: URL must be in the format 'imd://host:port'")
    
    # Check if the format is correct
    parts = filename.split("imd://")[1].split(":")
    if len(parts) == 2:
        host = parts[0] 
        try:
            port = int(parts[1])
            return (host, port)
        except ValueError:
            raise ValueError("IMDReader: Port must be an integer")
    else:
        raise ValueError("IMDReader: URL must be in the format 'imd://host:port'")

class StreamReaderBase(ReaderBase):

    def __init__(self, filename, convert_units=True, **kwargs):
        super(StreamReaderBase, self).__init__(
            filename, convert_units=convert_units, **kwargs
        )
        self._init_scope = True
        self._reopen_called = False
        self._first_ts = None

    def _read_next_timestep(self):
        # No rewinding- to both load the first frame after  __init__
        # and access it again during iteration, we need to store first ts in mem
        if not self._init_scope and self._frame == -1:
            self._frame += 1
            # can't simply return the same ts again- transformations would be applied twice
            # instead, return the pre-transformed copy
            return self._first_ts

        ts = self._read_frame(self._frame + 1)

        if self._init_scope:
            self._first_ts = self.ts.copy()
            self._init_scope = False

        return ts

    @property
    def n_frames(self):
        """Changes as stream is processed unlike other readers"""
        raise RuntimeError(
            "{}: n_frames is unknown".format(self.__class__.__name__)
        )

    def __len__(self):
        raise RuntimeError(
            "{} has unknown length".format(self.__class__.__name__)
        )

    def next(self):
        """Don't rewind after iteration. When _reopen() is called,
        an error will be raised
        """
        try:
            ts = self._read_next_timestep()
        except (EOFError, IOError):
            # Don't rewind here like we normally would
            raise StopIteration from None
        else:
            for auxname, reader in self._auxs.items():
                ts = self._auxs[auxname].update_ts(ts)

            ts = self._apply_transformations(ts)

        return ts

    def rewind(self):
        """Raise error on rewind"""
        raise RuntimeError(
            "{}: Stream-based readers can't be rewound".format(
                self.__class__.__name__
            )
        )

    # Incompatible methods
    def copy(self):
        raise NotImplementedError(
            "{} does not support copying".format(self.__class__.__name__)
        )

    def _reopen(self):
        if self._reopen_called:
            raise RuntimeError(
                "{}: Cannot reopen stream".format(self.__class__.__name__)
            )
        self._frame = -1
        self._reopen_called = True

    def __getitem__(self, frame):
        """Return the Timestep corresponding to *frame*.

        If `frame` is a integer then the corresponding frame is
        returned. Negative numbers are counted from the end.

        If frame is a :class:`slice` then an iterator is returned that
        allows iteration over that part of the trajectory.

        Note
        ----
        *frame* is a 0-based frame index.
        """
        if isinstance(frame, slice):
            _, _, step = self.check_slice_indices(
                frame.start, frame.stop, frame.step
            )
            if step is None:
                return FrameIteratorAll(self)
            else:
                return StreamFrameIteratorSliced(self, step)
        else:
            raise TypeError(
                "Streamed trajectories must be an indexed using a slice"
            )

    def check_slice_indices(self, start, stop, step):
        if start is not None:
            raise ValueError(
                "{}: Cannot expect a start index from a stream, 'start' must be None".format(
                    self.__class__.__name__
                )
            )
        if stop is not None:
            raise ValueError(
                "{}: Cannot expect a stop index from a stream, 'stop' must be None".format(
                    self.__class__.__name__
                )
            )
        if step is not None:
            if isinstance(step, numbers.Integral):
                if step < 1:
                    raise ValueError(
                        "{}: Cannot go backwards in a stream, 'step' must be > 0".format(
                            self.__class__.__name__
                        )
                    )
            else:
                raise ValueError(
                    "{}: 'step' must be an integer".format(
                        self.__class__.__name__
                    )
                )

        return start, stop, step

    def __getstate__(self):
        raise NotImplementedError(
            "{} does not support pickling".format(self.__class__.__name__)
        )

    def __setstate__(self, state: object):
        raise NotImplementedError(
            "{} does not support pickling".format(self.__class__.__name__)
        )

    def __repr__(self):
        return (
            "<{cls} {fname} with continuous stream of {natoms} atoms>"
            "".format(
                cls=self.__class__.__name__,
                fname=self.filename,
                natoms=self.n_atoms,
            )
        )


class StreamFrameIteratorSliced(FrameIteratorBase):

    def __init__(self, trajectory, step):
        super().__init__(trajectory)
        self._step = step

    def __iter__(self):
        # Calling reopen tells reader
        # it can't be reopened again
        self.trajectory._reopen()
        return self

    def __next__(self):
        try:
            # Burn the timesteps until we reach the desired step
            # Don't use next() to avoid unnecessary transformations
            while self.trajectory._frame + 1 % self.step != 0:
                self.trajectory._read_next_timestep()
        except (EOFError, IOError):
            # Don't rewind here like we normally would
            raise StopIteration from None

        return self.trajectory.next()

    def __len__(self):
        raise RuntimeError(
            "{} has unknown length".format(self.__class__.__name__)
        )

    def __getitem__(self, frame):
        raise RuntimeError("Sliced iterator does not support indexing")

    @property
    def step(self):
        return self._step

class IMDReader(StreamReaderBase):
    """
    Reader for IMD protocol packets.

    Parameters
    ----------
    filename : a string of the form "host:port" where host is the hostname
        or IP address of the listening GROMACS server and port
        is the port number.
    n_atoms : int (optional)
        number of atoms in the system. defaults to number of atoms
        in the topology. don't set this unless you know what you're doing.
    kwargs : dict (optional)
        keyword arguments passed to the constructed :class:`IMDClient`
    """

    format = "IMD"
    one_pass = True

    @store_init_arguments
    def __init__(
        self,
        filename,
        convert_units=True,
        n_atoms=None,
        **kwargs,
    ):
        super(IMDReader, self).__init__(filename, **kwargs)

        logger.debug("IMDReader initializing")

        if n_atoms is None:
            raise ValueError("IMDReader: n_atoms must be specified")
        self.n_atoms = n_atoms

        host, port = parse_host_port(filename)

        # This starts the simulation
        self._imdclient = IMDClient(host, port, n_atoms, **kwargs)

        imdsinfo = self._imdclient.get_imdsessioninfo()
        # NOTE: after testing phase, fail out on IMDv2

        self.ts = self._Timestep(
            self.n_atoms,
            positions=imdsinfo.positions,
            velocities=imdsinfo.velocities,
            forces=imdsinfo.forces,
            **self._ts_kwargs,
        )

        self._frame = -1

        try:
            self._read_next_timestep()
        except StopIteration:
            raise RuntimeError("IMDReader: No data found in stream")

    def _read_frame(self, frame):

        try:
            imdf = self._imdclient.get_imdframe()
        except EOFError as e:
            raise e

        self._frame = frame
        self._load_imdframe_into_ts(imdf)

        logger.debug(f"IMDReader: Loaded frame {self._frame}")
        return self.ts

    def _load_imdframe_into_ts(self, imdf):
        self.ts.frame = self._frame
        if imdf.time is not None:
            self.ts.time = imdf.time
            # NOTE: timestep.pyx "dt" method is suspicious bc it uses "new" keyword for a float
            self.ts.data["dt"] = imdf.dt
            self.ts.data["step"] = imdf.step
        if imdf.energies is not None:
            self.ts.data.update(
                {k: v for k, v in imdf.energies.items() if k != "step"}
            )
        if imdf.box is not None:
            self.ts.dimensions = core.triclinic_box(*imdf.box)
        if imdf.positions is not None:
            # must call copy because reference is expected to reset
            # see 'test_frame_collect_all_same' in MDAnalysisTests.coordinates.base
            self.ts.positions = imdf.positions
        if imdf.velocities is not None:
            self.ts.velocities = imdf.velocities
        if imdf.forces is not None:
            self.ts.forces = imdf.forces

    @staticmethod
    def _format_hint(thing):
        try:
            parse_host_port(thing)
        except:
            return False
        return True

    def close(self):
        """Gracefully shut down the reader. Stops the producer thread."""
        logger.debug("IMDReader close() called")
        self._imdclient.stop()
        # NOTE: removeme after testing
        logger.debug("IMDReader shut down gracefully.")
