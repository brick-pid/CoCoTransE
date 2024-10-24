main() {
    try{
    let correct = ArrayList<Bool>(
            !pairsSumToZero(ArrayList<Int64>(1, 3, 5, 0)),
            !pairsSumToZero(ArrayList<Int64>(1, 3, -2, 1)),
            !pairsSumToZero(ArrayList<Int64>(1, 2, 3, 7)),
            pairsSumToZero(ArrayList<Int64>(2, 4, -5, 3, 5, 7)),
            !pairsSumToZero(ArrayList<Int64>([1])),
            pairsSumToZero(ArrayList<Int64>(-3, 9, -1, 3, 2, 30)),
            pairsSumToZero(ArrayList<Int64>(-3, 9, -1, 3, 2, 31)),
            !pairsSumToZero(ArrayList<Int64>(-3, 9, -1, 4, 2, 30)),
            !pairsSumToZero(ArrayList<Int64>(-3, 9, -1, 4, 2, 31))
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