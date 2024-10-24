main() {
    try{
    let correct = ArrayList<Bool>(
            validDate("03-11-2000" ) == true,
            validDate("15-01-2012" ) == false,
            validDate("04-0-2040" ) == false,
            validDate("06-04-2020" ) == true,
            validDate("01-01-2007" ) == true,
            validDate("03-32-2011" ) == false,
            validDate("" ) == false,
            validDate("04-31-3000" ) == false,
            validDate("06-06-2005" ) == true,
            validDate("21-31-2000" ) == false,
            validDate("04-12-2003" ) == true,
            validDate("04122003" ) == false,
            validDate("20030412" ) == false,
            validDate("2003-04" ) == false,
            validDate("2003-04-12" ) == false,
            validDate("04-2003" ) == false
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}