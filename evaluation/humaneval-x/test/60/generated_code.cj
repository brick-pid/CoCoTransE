main() {
    try{
    let correct = ArrayList<Bool>(
            sumToN(1) == 1,
            sumToN(6) == 21,
            sumToN(11) == 66,
            sumToN(30) == 465,
            sumToN(100) == 5050
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