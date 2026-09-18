"""One typed word from the terminal, shared by the two legs.

Both entries ask the operator to type a word at several prompts (`retry`, `go`,
`shift`, `quit`, `home`). Keeping the key handling in one place means they behave
the same everywhere:

* Backspace and the Delete key (`ESC [ 3 ~`) erase the last character *on screen*,
  not just in the buffer -- the old loops only dropped it from the buffer, so a
  mistyped word looked uneditable;
* Ctrl+U clears the whole word;
* arrow keys / Home / End are consumed as a complete escape sequence and ignored,
  so they neither stop the session nor leak into the word;
* a lone Escape still stops the session (the caller turns ``('stop',)`` into its
  own stop path).
"""
import os
import select
import time


def word_key(fd, poll=.02):
    """Classify one terminal key press for a typed word.

    Returns None when nothing arrived within ``poll`` seconds, else one of
    ``('char', text)`` / ``('backspace',)`` / ``('clear',)`` / ``('enter',)`` /
    ``('stop',)`` / ``('eof',)`` / ``('ignored',)``.
    """
    if not select.select([fd], [], [], poll)[0]:
        return None
    key = os.read(fd, 1)
    if key == b'\x1b':
        sequence = b''
        while select.select([fd], [], [], .03)[0]:
            byte = os.read(fd, 1)
            sequence += byte
            if len(sequence) == 1 and byte not in (b'[', b'O'):
                break                    # Alt+key or a stray byte: nothing to finish
            if len(sequence) > 1 and 0x40 <= byte[0] <= 0x7e:
                break                    # CSI/SS3 final byte ends the sequence
            if len(sequence) >= 8:
                break
        if not sequence:
            return ('stop',)
        return ('backspace',) if sequence in (b'[3~', b'[2~') else ('ignored',)
    if key in (b' ', b'\x03'):
        return ('stop',)
    if key in (b'', b'\x04'):
        return ('eof',)
    if key in (b'\r', b'\n'):
        return ('enter',)
    if key in (b'\x7f', b'\x08'):
        return ('backspace',)
    if key == b'\x15':
        return ('clear',)
    if key[0] < 0x20:
        return ('ignored',)
    return ('char', key.decode(errors='ignore'))


def type_word(fd, poll, prompt=None, deadline=None):
    """Read one typed word, echoing it while it is edited.

    ``poll`` runs before every key for the usual per-loop work (arm ticks,
    preview, lock checks) and may raise to leave the session. Returns
    ``('word', text)``, ``('stop',)``, ``('eof',)`` or ``('timeout',)``.
    """
    if prompt is not None:
        print(prompt, flush=True)
    pending = ''
    while True:
        poll()
        if deadline is not None and time.monotonic() > deadline:
            return ('timeout', '')
        key = word_key(fd)
        if key is None:
            continue
        kind = key[0]
        if kind == 'stop':
            return ('stop', '')
        if kind == 'eof':
            return ('eof', '')
        if kind == 'enter':
            print()
            return ('word', pending)
        if kind == 'backspace':
            if pending:
                pending = pending[:-1]
                print('\b \b', end='', flush=True)
            continue
        if kind == 'clear':
            print('\b \b'*len(pending), end='', flush=True)
            pending = ''
            continue
        if kind == 'char':
            pending += key[1]
            print(key[1], end='', flush=True)
