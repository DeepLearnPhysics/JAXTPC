"""
Entry naming and identity attributes of production files (sensor/step/hits).

Format 2 (``config.attrs['format_version'] == 2``)
-------------------------------------------------
* One group per event, named ``entry_NNNNNN``: NNNNNN is the entry's position
  in the file (0-based, input order), nothing more.  An event that was skipped
  (see the overflow log) leaves a gap.
* What the event *is* lives in attributes of the entry group:

  ``source_index``      which input file it came from: an index into
                        ``config/source_files``
  ``source_event_idx``  its index (row) within that input file
  ``event_id``          its identifier.  With dprod inputs (an edep-sim file
                        carrying ``/dprod/events``) this is the event id that
                        is unique within the production campaign; otherwise
                        as before (``--edepsim-ids``: edep-sim's own event id;
                        else ``event_id_offset + source_event_idx``)
  ``job_id``            dprod inputs only: the production job that generated
                        the event (edep-sim's run id)
  ``job_event``         dprod inputs only: the event's number within that job

* ``config/source_files`` lists the input file(s); ``config.attrs['source_file']``
  is still written when a file has a single input.

Format 1 (no ``format_version``): groups ``event_NNN`` (NNN = index within
the file), one input per file named by ``config.attrs['source_file']``.

Readers should not rebuild group names themselves: use :func:`entry_keys` to
list a file's entries and :func:`find_entry` to get entry *i* of either format.
"""

FORMAT_VERSION = 2
ENTRY_PREFIX = 'entry_'
LEGACY_PREFIX = 'event_'

#: Entry attributes carried over from dprod's /dprod/events table.
DPROD_ID_ATTRS = ('event_id', 'job_id', 'job_event')


def entry_key(index):
    """Group name of entry *index* in a format-2 file."""
    return f'{ENTRY_PREFIX}{int(index):06d}'


def legacy_key(index):
    """Group name of event *index* in a format-1 file."""
    return f'{LEGACY_PREFIX}{int(index):03d}'


def _index(key):
    for prefix in (ENTRY_PREFIX, LEGACY_PREFIX):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return int(key[len(prefix):])
    return None


def entry_keys(f):
    """The file's entry (event) group names, in entry order; either format."""
    keys = [(i, k) for k, i in ((k, _index(k)) for k in f.keys()) if i is not None]
    return [k for _, k in sorted(keys)]


def find_entry(f, index):
    """Group name of entry *index* (format 2) or event *index* (format 1).

    Raises KeyError if the file has neither.
    """
    for k in (entry_key(index), legacy_key(index)):
        if k in f:
            return k
    raise KeyError(f'no entry {index} in {getattr(f, "filename", f)!r} '
                   f'(neither {entry_key(index)} nor {legacy_key(index)})')


def source_files(f):
    """The input file(s) a production file was made from, as a list of str."""
    cfg = f.get('config')
    if cfg is None:
        return []
    if 'source_files' in cfg:
        return [s.decode() if isinstance(s, bytes) else str(s)
                for s in cfg['source_files'][()]]
    src = cfg.attrs.get('source_file')
    if src is None:
        return []
    return [src.decode() if isinstance(src, bytes) else str(src)]
