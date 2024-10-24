main() {
    try{
    let correct = ArrayList<Bool>(
            getOddCollatz(14) == ArrayList<Int64>(1, 5, 7, 11, 13, 17),
            getOddCollatz(5) == ArrayList<Int64>(1, 5),
            getOddCollatz(12) == ArrayList<Int64>(1, 3, 5),
            getOddCollatz(1) == ArrayList<Int64>([1])
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