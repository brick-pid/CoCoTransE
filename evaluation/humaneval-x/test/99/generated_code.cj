main() {
    try{
    let correct = ArrayList<Bool>(
            countUpper("10") == 10,
            countUpper("14.5") == 15,
            countUpper("-15.5") == -16,
            countUpper("15.3") == 15,
            countUpper("0") == 0
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