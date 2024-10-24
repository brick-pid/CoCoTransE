main() {
    try{
    let correct = ArrayList<Bool>(
            !triplesSumToZero(ArrayList<Int64>(1, 3, 5, 0)),
            !triplesSumToZero(ArrayList<Int64>(1, 3, 5, -1)),
            triplesSumToZero(ArrayList<Int64>(1, 3, -2, 1)),
            !triplesSumToZero(ArrayList<Int64>(1, 2, 3, 7)),
            !triplesSumToZero(ArrayList<Int64>(1, 2, 5, 7)),
            triplesSumToZero(ArrayList<Int64>(2, 4, -5, 3, 9, 7)),
            !triplesSumToZero(ArrayList<Int64>([1])),
            !triplesSumToZero(ArrayList<Int64>(1, 3, 5, -100)),
            !triplesSumToZero(ArrayList<Int64>(100, 3, 5, -100))
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