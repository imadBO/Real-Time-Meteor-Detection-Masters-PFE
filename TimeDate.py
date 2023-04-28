import datetime
from datetime import datetime as dt
from math import modf

class TimeDate:

    class Date:
        def __init__(self):
            self.year = 0
            self.month = 0
            self.day = 0
            self.hours = 0
            self.minutes = 0
            self.seconds = 0.0

    @staticmethod
    def localDateTime(pt, format):
        s = ''
        datetime_ss = dt.strftime(pt,format)
        return datetime_ss

    @staticmethod
    def getCurrentDateYYYYMMDD():
        return dt.utcnow().strftime("%Y,%m,%d")

    @staticmethod
    def gregorianToJulian(date):
        if date.month == 1 or date.month == 2:
            date.year = date.year - 1
            date.month = date.month + 12

        f0, A = modf(date.year/100)
        f1, A1 = modf(A/4)
        B = 2 - A + A1
        f2, C = modf(365.25*date.year)
        f3, D = modf(30.6001*(date.month+1))

        val = B + C + D + date.day + 1720994.5
        return val

    @staticmethod
    def julianCentury(julianDate):
        julianCentury = (julianDate - 2451545.0) / 36525.0
        return julianCentury

    @staticmethod
    def hmsToHdecimal(H, M, S):
        decimalHours = H + (M / 60.0) + (S / 3600.0)
        return decimalHours

    @staticmethod
    def HdecimalToHMS(val):
        H = int(val)
        M = int((val - H) * 60)
        S = int((val - H - (M / 60.0)) * 3600)
        return [H, M, S]

    @staticmethod
    def localSideralTime_2(julianCentury, gregorianH, gregorianMin, gregorianS, longitude):
        JD0 = 2451545.0
        siderealTime = 280.16 + 360.9856235 * (TimeDate.gregorianToJulian(TimeDate.Date(year=0, month=0, day=0, hours=gregorianH, minutes=gregorianMin, seconds=gregorianS)) - JD0) + longitude
        while siderealTime < 0:
            siderealTime += 360.0
        while siderealTime > 360:
            siderealTime -= 360.0
        return siderealTime

    @staticmethod
    def localSideralTime_1(JD0, gregorianH, gregorianMin, gregorianS):
        siderealTime = 280.16 + 360.9856235 * (TimeDate.gregorianToJulian(TimeDate.Date(year=0, month=0, day=0, hours=gregorianH, minutes=gregorianMin, seconds=gregorianS)) - JD0)
        while siderealTime < 0:
            siderealTime += 360.0
        while siderealTime > 360:
            siderealTime -= 360.0
        return siderealTime

    @staticmethod
    def splitStringToInt(date):
        date_parts = date

    @staticmethod
    def getYYYYMMDDfromDateString(date):
        dt = datetime.datetime.fromisoformat(date)
        return dt.strftime('%Y%m%d')

    @staticmethod
    def getYYYYMMDD(date):
        return date.year * 10000 + date.month * 100 + date.day

    @staticmethod
    def getIntVectorFromDateString(date):
        dt = datetime.datetime.fromisoformat(date)
        return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]

    @staticmethod
    def getYYYYMMDDThhmmss(date):
        dt = datetime.datetime.fromisoformat(date)
        return dt.strftime('%Y%m%dT%H%M%S')

    @staticmethod
    def getYYYYMMDDThhmmss(date):
        return date.year * 100000000 + date.month * 1000000 + date.day * 10000 + date.hours * 100 + date.minutes

    @staticmethod
    def splitIsoExtendedDate(date):
        dt = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')
        date_obj = TimeDate.Date()
        date_obj.year = dt.year
        date_obj.month = dt.month
        date_obj.day = dt.day
        date_obj.hours = dt.hour
        date_obj.minutes = dt.minute
        date_obj.seconds = dt.second + dt.microsecond / 1000000.0
        return date_obj

    @staticmethod
    def getIsoExtendedFormatDate(date):
        isoExtendedDate = "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:06.3f}".format(date.year, date.month, date.day, date.hours, date.minutes, date.seconds)
        return isoExtendedDate

        # return date.strftime('%Y-%m-%dT%H:%M:%S.%f')

    @staticmethod
    def secBetweenTwoDates(d1, d2):
        delta = datetime.datetime(d2.year, d2.month, d2.day, d2.hours, d2.minutes, int(d2.seconds)) - \
                datetime.datetime(d1.year, d1.month, d1.day, d1.hours, d1.minutes, int(d1.seconds))
        return delta.total_seconds()