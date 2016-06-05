from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%m-%d-%Y")
    d2 = datetime.strptime(d2, "%m-%d-%Y")
    print (abs((d2 - d1).days))

days_between("6-17-2013", "10-25-2013")