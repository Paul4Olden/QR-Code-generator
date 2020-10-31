import collections

from bitarray import bitarray
import re

WHITE = False
BLACK = True
FUNCTIONAL_PATTERN = 'FP'
DATA_PATTERN = 'DP'

QRCODE_CAPACITY = [
    -1, 208, 359, 567, 807, 1079, 1383, 1568, 1936, 2336, 2768, 3232, 3728, 4256, 4651, 5243, 5867, 6523, 7211, 7931,
    8683, 9252, 10068, 10916, 11796, 12708, 13652, 14628, 15371, 16411, 17483, 18587, 19723, 20891, 22091, 23008, 24272,
    25568, 26896, 28256, 29648
]

REMINDER_BITS = [
    -1, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,
    0, 0, 0
]

MODE = {
    'NUMERIC': 1,
    'ALPHA_NUMERIC': 2,
    'BYTE': 4,
    'KANJI': 8,
    'ECI': 7
}

MASK_PATTERNS = {
    0: lambda x, y: (x + y) % 2,
    1: lambda x, y: y % 2,
    2: lambda x, y: x % 3,
    3: lambda x, y: (x + y) % 3,
    4: lambda x, y: (x // 3 + y // 2) % 2,
    5: lambda x, y: x * y % 2 + x * y % 3,
    6: lambda x, y: (x * y % 2 + x * y % 3) % 2,
    7: lambda x, y: ((x + y) % 2 + x * y % 3) % 2,
}

ERROR_CORRECTION_LEVELS = {
    0: [
        (-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30,
         30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
        (-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17,
         18, 19, 19, 20, 21, 22, 24, 25)
    ],
    1: [
        (-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28,
         28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
        (-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29,
         31, 33, 35, 37, 38, 40, 43, 45, 47, 49)
    ],
    2: [
        (-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30,
         30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
        (-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38,
         40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68)
    ],
    3: [
        (-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30,
         30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
        (-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45,
         48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81)
    ]
}

REGEX = {
    MODE['NUMERIC']: re.compile(r"[0-9]"),
    MODE['ALPHA_NUMERIC']: re.compile(r"[0-9A-Z *$%+./:-]")
    # TODO: ADD KANJI MODE REGEX
}

alphaNumericValues = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'


class QrBytearray:
    @staticmethod
    def get_version(data_bit_length: int, ecc_level: int) -> int:
        length = data_bit_length + ecc_level + 8

        for el in QRCODE_CAPACITY:
            if length <= el:
                return QRCODE_CAPACITY.index(el)

    @staticmethod
    def get_mode(data: str) -> int:
        if re.fullmatch(re.compile(r"[0-9]"), data):
            return MODE['NUMERIC']

        elif re.fullmatch(re.compile(r"[0-9A-Z *$%+./:-]"), data):
            return MODE['ALPHA_NUMERIC']

        else:
            return MODE['BYTE']

    def _add_mode(self, mode: int):
        out = bin(mode)[2:].zfill(4)
        self.extend(out)

    def _add_counter(self, count: int, mode: list, version: int):
        if 1 <= version <= 9:
            out = bin(count)[2:].zfill(mode[0])
            self.extend(out)
        elif 10 <= version <= 26:
            out = bin(count)[2:].zfill(mode[1])
            self.extend(out)
        elif 27 <= version <= 40:
            out = bin(count)[2:].zfill(mode[2])
            self.extend(out)
        else:
            raise ValueError('Version is not correct')

    def _add_terminator(self):
        unfilled_bits = self._get_unfilled_bits()
        if unfilled_bits >= 4:
            out = bin(0)[2:].zfill(4)
            self.extend(out)

        else:
            out = bin(0)[2:].zfill(unfilled_bits)
            self.extend(out)

    def _add_bit_padding(self):
        if len(self.keyword) % 8 != 0:
            self.extend(bin(0)[2:].zfill(len(self.keyword) % 8))

    def _add_bytes_padding(self):
        unfilled_bits = self._get_unfilled_bits()

        def repeat_string(a_string, target_length):
            number_of_repeats = target_length // len(a_string) + 1
            a_string_repeated = a_string * number_of_repeats
            a_string_repeated_to_target = a_string_repeated[:target_length]
            return a_string_repeated_to_target

        padding_value = '1110110000010001'

        if unfilled_bits > 0:
            out = ''.join(repeat_string(padding_value, unfilled_bits))
            self.extend(out)

    def _get_unfilled_bits(self):
        version = self.version
        ecc_level = self.ecc_level
        return QRCODE_CAPACITY[version] - (
                    len(self.keyword) + REMINDER_BITS[version] + ERROR_CORRECTION_LEVELS[ecc_level][0][version] * 8)

    def __init__(self, data: bytearray, data_len: int, version: int, mode: int, counter: list, ecc_level):
        self.keyword = ''
        self.version = version
        self.ecc_level = ecc_level

        self._add_mode(mode)
        self._add_counter(data_len, counter, self.version)

        self.from_bytes(data)

        self._add_terminator()
        self._add_bit_padding()
        self._add_bytes_padding()
        print('s')

    def extend(self, data: str):
        reg = re.compile('[0-1]+')
        for char in data:
            if re.fullmatch(reg, char):
                self.keyword += char

            else:
                raise ValueError('Argument must contain 0 and 1 only got {} instead.'.format(char))

    def from_bytes(self, data: bytearray):
        for byte in data:
            self.extend(bin(byte)[2:].zfill(8))

    def to_bytes(self) -> bytearray:
        if len(self.keyword) % 8 == 0:
            data = [self.keyword[i:i + 8] for i in range(0, len(self.keyword), 8)]
            data = [int(i, 2) for i in data]
            out = bytearray(data)
            return out
        raise ValueError('Data is not converted to bytes')

    @staticmethod
    def encode_byte(data: str, version: int, ecc_level):
        """ Return bytearray object representing data ready for QR code using byte mode """
        counter = [8, 16, 16]
        _data = bytearray(data.encode())

        if version == -1:
            version = QrBytearray.get_version(len(data) * 8, ecc_level)

        return QrBytearray(
            _data, len(data), version, QrBytearray.get_mode(data), counter, ecc_level
        ).to_bytes()

    @staticmethod
    def encode_alphanumeric(data: str, version: int, ecc_level: str) -> bytearray:
        """ Return bytearray object representing data ready for QR code using alphanumeric mode """
        _data = ''
        counter = [9, 11, 13]

        chars = [data[i:i + 2] for i in range(0, len(data), 2)]
        for x in chars:
            if len(x) == 2:
                _data.join(bin(alphaNumericValues.index(x[0]))[2:].zfill(7) * 45 + bin(alphaNumericValues.index(x[1]))[
                                                                                   2:].zfill(7))
            else:
                _data.join(bin(alphaNumericValues.index(x[0]))[2:].zfill(7))

        return QrBytearray(
            bitarray(_data),
            len(data),
            version,
            QrBytearray.get_mode(data),
            counter,
            ecc_level
        ).tobytes()

    @staticmethod
    def encode_numeric(data: str, version: int, ecc_level: str) -> bytearray:
        counter = [10, 12, 14]
        numbers = [data[i:i + 3] for i in range(0, len(data), 3)]
        out_bytes = list(map(lambda x: bin(int(x))[2:].zfill(10), numbers))

        return QrBytearray(
            bitarray(out_bytes),
            len(data),
            version,
            QrBytearray.get_mode(data),
            counter,
            ecc_level
        ).tobytes()


class ReedSolo:
    def init_tables(self, prim=0x11d, generator=2, c_exp=8):
        from array import array

        def _bytearray(obj=0, encoding="latin-1"):
            if isinstance(obj, str):
                obj = obj.encode(encoding)
                if isinstance(obj, str):
                    obj = [ord(chr) for chr in obj]
                elif isinstance(obj, bytes):
                    obj = [int(chr) for chr in obj]
                else:
                    raise (ValueError, "Type of object not recognized!")
            elif isinstance(obj, int):
                obj = [0] * obj
            return array("i", obj)

        field_charac = int(2 ** c_exp - 1)
        gf_exp = _bytearray(
            field_charac * 2)
        gf_log = _bytearray(field_charac + 1)

        x = 1
        for i in range(
                field_charac):
            gf_exp[i] = x
            gf_log[x] = i
            x = self.gf_mult_noLUT(x, generator, prim, field_charac + 1)

        for i in range(field_charac, field_charac * 2):
            gf_exp[i] = gf_exp[i - field_charac]

        return [gf_log, gf_exp, field_charac]

    def gf_mult_noLUT(self, x, y, prim=0, field_charac_full=256, carryless=True):
        r = 0
        while y:
            if y & 1: r = r ^ x if carryless else r + x
            y = y >> 1
            x = x << 1
            if prim > 0 and x & field_charac_full: x = x ^ prim

        return r

    def gf_power(self, x, power):
        return self.gf_exp[(self.gf_log[x] * power) % self.field_character]

    def chunk(self, data, chunksize):
        for i in range(0, len(data), chunksize):
            chunk = data[i:i + chunksize]
            yield chunk

    def gf_poly_mul(self, p, q):
        r = bytearray(len(p) + len(q) - 1)
        lp = [self.gf_log[p[i]] for i in range(len(p))]

        for j in range(len(q)):
            qj = q[j]
            if qj != 0:
                lq = self.gf_log[qj]
                for i in range(len(p)):
                    if p[i] != 0:
                        r[i + j] ^= self.gf_exp[lp[i] + lq]
        return r

    def rs_generator_poly(self, nsym_gp):
        out_gen = bytearray([1])
        for x in range(nsym_gp):
            out_gen = self.gf_poly_mul(out_gen, [1, self.gf_power(2, x)])
        return out_gen

    def rs_encode_msg(self, msg_in, nsym):
        gen = self.rs_generator_poly(nsym)

        msg_in = bytearray(msg_in)
        msg_out = bytearray(msg_in) + bytearray(len(gen) - 1)

        lgen = bytearray([self.gf_log[gen[j]] for j in range(len(gen))])

        for i in range(len(msg_in)):
            coef = msg_out[i]

            if coef != 0:
                lcoef = self.gf_log[coef]

                for j in range(1, len(gen)):
                    msg_out[i + j] ^= self.gf_exp[lcoef + lgen[j]]

        msg_out[:len(msg_in)] = msg_in
        return msg_out

    def reed_solomon_encode(self, data: bytearray, nsym: int) -> bytearray:
        """ Return bytearray object representing reed-solomon error correction code for given data """
        nsize = 255
        gf_log = bytearray(256)
        gf_exp = bytearray([1] * 512)
        field_character = int(2 ** 8 - 1)

        if isinstance(data, str):
            data = bytearray(data)
        enc = bytearray()
        for chunk in self.chunk(data, nsize - nsym):
            enc.extend(self.rs_encode_msg(chunk, nsym))
        return enc

    def __init__(self):
        self.gf_log, self.gf_exp, self.field_character = self.init_tables()


class QRGeneratorBase:
    def _get_bit(self, x: int, i: int) -> bool:
        return (x >> i) & 1 != 0

    def _create_qrarray(self, size: int):
        """Create square with maximal capacity"""
        self.matrix = [[False] * size] * size
        self.pattern_matrix = [[False] * size] * size

    def _draw_timing_pattern(self):
        for i in range(self.size):
            self._set_function_module(6, i, i % 2 == 0)
            self._set_function_module(i, 6, i % 2 == 0)

    def _draw_finder_pattern(self, x, y) -> None:
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                xx, yy = x + dx, y + dy
                if (0 <= xx < self.size) and (0 <= yy < self.size):
                    # Chebyshev/infinity norm
                    self._set_function_module(xx, yy, max(abs(dx), abs(dy)) not in (2, 4))

    def _draw_alignment_pattern(self, x, y) -> None:
        """Draws a 5*5 alignment pattern, with the center module
        at (x, y). All modules must be in bounds."""
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                self._set_function_module(x + dx, y + dy, max(abs(dx), abs(dy)) != 1)

    def _set_function_module(self, x: int, y: int, isblack: bool) -> None:
        assert type(isblack) is bool
        self.matrix[y][x] = isblack
        self.pattern_matrix[y][x] = True

    def _draw_format_bits(self, mask) -> None:
        data = self.ecc_level.formatbits << 3 | mask
        rem = data
        for _ in range(10):
            rem = (rem << 1) ^ ((rem >> 9) * 0x537)
        bits = (data << 10 | rem) ^ 0x5412
        assert bits >> 15 == 0

        # Draw first copy
        for i in range(0, 6):
            self._set_function_module(8, i, self._get_bit(bits, i))
        self._set_function_module(8, 7, self._get_bit(bits, 6))
        self._set_function_module(8, 8, self._get_bit(bits, 7))
        self._set_function_module(7, 8, self._get_bit(bits, 8))
        for i in range(9, 15):
            self._set_function_module(14 - i, 8, self._get_bit(bits, i))

        # Draw second copy
        for i in range(0, 8):
            self._set_function_module(self.size - 1 - i, 8, self._get_bit(bits, i))
        for i in range(8, 15):
            self._set_function_module(8, self.size - 15 + i, self._get_bit(bits, i))
        self._set_function_module(8, self.size - 8, True)  # Always black

    def _draw_version(self):
        if self.version < 7:
            return

        rem = self.version
        for _ in range(12):
            rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
        bits = self.version << 12 | rem
        assert bits >> 18 == 0

        for i in range(18):
            bit = self._get_bit(bits, i)
            a = self.size - 11 + i % 3
            b = i // 3
            self._set_function_module(a, b, bit)
            self._set_function_module(b, a, bit)

    def _get_alignment_pattern_position(self) -> list:
        ver = self.version
        if ver == 1:
            return []
        else:
            numalign = ver // 7 + 2
            step = 26 if (ver == 32) else \
                (ver * 4 + numalign * 2 + 1) // (numalign * 2 - 2) * 2
            result = [(self.size - 7 - i * step) for i in range(numalign - 1)] + [6]
            return list(reversed(result))

    def _draw_function_patterns(self):
        raise NotImplemented()

    def _apply_mask(self, mask: int) -> None:
        """XORs the codeword modules in this QR Code with the given mask pattern.
        The function modules must be marked and the codeword bits must be drawn
        before masking. Due to the arithmetic of XOR, calling applyMask() with
        the same mask value a second time will undo the mask. A final well-formed
        QR Code needs exactly one (not zero, two, etc.) mask applied."""
        if not (0 <= mask <= 7):
            raise ValueError("Mask value out of range")
        masker = MASK_PATTERNS[mask]
        for y in range(self.size):
            for x in range(self.size):
                self.matrix[y][x] ^= (masker(x, y) == 0) and (not self.pattern_matrix[y][x])

    def _get_penalty_score(self) -> int:
        result = 0
        size = self.size
        modules = self.matrix

        for y in range(size):
            runcolor = False
            runx = 0
            runhistory = collections.deque([0] * 7, 7)
            for x in range(size):
                if modules[y][x] == runcolor:
                    runx += 1
                    if runx == 5:
                        result += QRGeneratorBase._PENALTY_N1
                    elif runx > 5:
                        result += 1
                else:
                    self._finder_penalty_add_history(runx, runhistory)
                    if not runcolor:
                        result += self._finder_penalty_count_patterns(runhistory) * QRGeneratorBase._PENALTY_N3
                    runcolor = modules[y][x]
                    runx = 1
            result += self._finder_penalty_terminate_and_count(runcolor, runx, runhistory) * QRGeneratorBase._PENALTY_N3

        for x in range(size):
            runcolor = False
            runy = 0
            runhistory = collections.deque([0] * 7, 7)
            for y in range(size):
                if modules[y][x] == runcolor:
                    runy += 1
                    if runy == 5:
                        result += QRGeneratorBase._PENALTY_N1
                    elif runy > 5:
                        result += 1
                else:
                    self._finder_penalty_add_history(runy, runhistory)
                    if not runcolor:
                        result += self._finder_penalty_count_patterns(runhistory) * QRGeneratorBase._PENALTY_N3
                    runcolor = modules[y][x]
                    runy = 1
            result += self._finder_penalty_terminate_and_count(runcolor, runy, runhistory) * QRGeneratorBase._PENALTY_N3

        for y in range(size - 1):
            for x in range(size - 1):
                if modules[y][x] == modules[y][x + 1] == modules[y + 1][x] == modules[y + 1][x + 1]:
                    result += QRGeneratorBase._PENALTY_N2

        black = sum((1 if cell else 0) for row in modules for cell in row)
        total = size ** 2
        k = (abs(black * 20 - total * 10) + total - 1) // total - 1
        result += k * QRGeneratorBase._PENALTY_N4
        return result

    def _finder_penalty_count_patterns(self, runhistory: collections.deque) -> int:
        n = runhistory[1]
        assert n <= self.size * 3
        core = n > 0 and (runhistory[2] == runhistory[4] == runhistory[5] == n) and runhistory[3] == n * 3
        return (1 if (core and runhistory[0] >= n * 4 and runhistory[6] >= n) else 0) \
               + (1 if (core and runhistory[6] >= n * 4 and runhistory[0] >= n) else 0)

    def _finder_penalty_terminate_and_count(self, currentruncolor: bool, currentrunlength: int,
                                            runhistory: collections.deque) -> int:
        if currentruncolor:
            self._finder_penalty_add_history(currentrunlength, runhistory)
            currentrunlength = 0
        currentrunlength += self.size
        self._finder_penalty_add_history(currentrunlength, runhistory)
        return self._finder_penalty_count_patterns(runhistory)

    def _finder_penalty_add_history(self, currentrunlength: int, runhistory: collections.deque) -> None:
        if runhistory[0] == 0:
            currentrunlength += self.size
        runhistory.appendleft(currentrunlength)

    def get_module(self, i: int, j: int) -> int:
        return self.matrix[i][j]

    def to_svg_str(self, border: int) -> str:
        if border < 0:
            raise ValueError("Border must be non-negative")
        parts = []
        for y in range(self.size):
            for x in range(self.size):
                if self.get_module(x, y):
                    parts.append("M{},{}h1v1h-1z".format(x + border, y + border))
        return """
                <?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
                <svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 {0} {0}" stroke="none">
                    <rect width="100%" height="100%" fill="#FFFFFF"/>
                    <path d="{1}" fill="#000000"/>
                </svg>
                """.format(self.size + border * 2, " ".join(parts))

    def _add_ecc(self, data: bytearray) -> list:
        version = self.version

        numblocks = ERROR_CORRECTION_LEVELS[self.ecc_level.ordinal][1][version]
        blockecclen = ERROR_CORRECTION_LEVELS[self.ecc_level.ordinal][0][version]
        rawcodewords = QRCODE_CAPACITY[version] // 8
        numshortblocks = numblocks - rawcodewords % numblocks
        shortblocklen = rawcodewords // numblocks

        blocks = []
        k = 0
        for i in range(numblocks):
            dat = data[k: k + shortblocklen - blockecclen + (0 if i < numshortblocks else 1)]
            k += len(dat)
            blocks.append(ReedSolo().reed_solomon_encode(dat, blockecclen))

        result = []
        for block in blocks:
            for byte in block:
                result.append(int(bin(byte)[2:].zfill(8), 2))
        
        return result

    def _draw_codewords(self, data: list) -> None:
        i = 0
        for right in range(self.size - 1, 0, -2):
            if right <= 6:
                right -= 1
            for vert in range(self.size):
                for j in range(2):
                    x = right - j
                    upward = (right + 1) & 2 == 0
                    y = (self.size - 1 - vert) if upward else vert
                    if not self.pattern_matrix[y][x] and i < len(data) * 8:
                        self.matrix[y][x] = self._get_bit(data[i >> 3], 7 - (i & 7))
                        i += 1

    class Ecc:
        def __init__(self, i: int, fb: int) -> None:
            self.ordinal = i
            self.formatbits = fb

        LOW: None
        MEDIUM: None
        QUARTILE: None
        HIGH: None

    def __init__(self, version: int, ecc) -> None:
        self.version = version
        self.size = self.version * 4 + 17
        self.ecc_level = ecc

        self.matrix = [[False] * self.size for _ in range(self.size)]
        self.pattern_matrix = [[False] * self.size for _ in range(self.size)]

    _PENALTY_N1 = 3
    _PENALTY_N2 = 3
    _PENALTY_N3 = 40
    _PENALTY_N4 = 10

    Ecc.LOW = Ecc(0, 1)
    Ecc.MEDIUM = Ecc(1, 0)
    Ecc.QUARTILE = Ecc(2, 3)
    Ecc.HIGH = Ecc(3, 2)


class QRCodeClassic(QRGeneratorBase):
    @staticmethod
    def generate_qr(data: str, ecc: QRGeneratorBase.Ecc, mask: int):
        if mask not in range(0,7):
            mask = -1
        version = QrBytearray.get_version(len(data) * 8, ecc.ordinal + 1)

        _data = QrBytearray.encode_byte(data, version, ecc.ordinal + 1)

        return QRCodeClassic(version, ecc, _data, mask)

    def _draw_function_patterns(self):
        self._draw_timing_pattern()

        self._draw_finder_pattern(3, 3)
        self._draw_finder_pattern(self.size - 4, 3)
        self._draw_finder_pattern(3, self.size - 4)

        alignment_position = self._get_alignment_pattern_position()
        num_align = len(alignment_position)
        skips = ((0, 0), (0, num_align - 1), (num_align - 1, 0))

        for i in range(num_align):
            for j in range(num_align):
                if (i, j) not in skips:
                    self._draw_alignment_pattern(alignment_position[i], alignment_position[j])

        self._draw_format_bits(0)
        self._draw_version()

    def __init__(self, version: int, ecc: QRGeneratorBase.Ecc, data: bytearray, mask: int) -> None:
        super().__init__(version, ecc)
        self._draw_function_patterns()
        allcodewords = self._add_ecc(data)
        self._draw_codewords(allcodewords)

        if mask == -1:
            minpenalty = 1 << 32
            for i in range(8):
                self._apply_mask(i)
                self._draw_format_bits(i)
                penalty = self._get_penalty_score()
                if penalty < minpenalty:
                    mask = i
                    minpenalty = penalty
                self._apply_mask(i)
        assert 0 <= mask <= 7
        self._apply_mask(mask)
        self._draw_format_bits(mask)

        self._mask = mask
        del self.pattern_matrix
