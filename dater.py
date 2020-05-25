from datetime import timedelta, date


def daterange(start_date, end_date):

    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def Dater(start,end):
    startyear, startmonth, startday = map(int, start.split('-'))
    endyear, endmonth, endday = map(int, end.split('-'))

    start_date = date(startyear, startmonth, startday)
    end_date = date(endyear, endmonth, endday)
    dates = []
    for single_date in daterange(start_date, end_date):
        dates.append(single_date.strftime("%Y-%m-%d"))
    return dates


