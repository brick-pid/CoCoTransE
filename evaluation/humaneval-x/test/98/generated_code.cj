main() {
    try{
    let correct = ArrayList<Bool>(
            countUpper("aBCdEf")  == 1,
            countUpper("abcdefg") == 0,
            countUpper("dBBE") == 0,
            countUpper("B")  == 0,
            countUpper("U")  == 1,
            countUpper("") == 0,
            countUpper("EEEE") == 2
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