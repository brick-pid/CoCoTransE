main() {
    try{
    let correct = ArrayList<Bool>(
            (decimalToBinary(0) == "db0db"),
            (decimalToBinary(32) == "db100000db"),
            (decimalToBinary(103) == "db1100111db"),
            (decimalToBinary(15) == "db1111db")
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