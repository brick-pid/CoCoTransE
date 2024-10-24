main() {
    try{
    let correct = ArrayList<Bool>(
            sortArray(ArrayList<Int64>(1, 5, 2, 3, 4)) == ArrayList<Int64>(1, 2, 4, 3, 5),
            sortArray(ArrayList<Int64>(-2, -3, -4, -5, -6)) == ArrayList<Int64>(-4, -2, -6, -5, -3),
            sortArray(ArrayList<Int64>(1, 0, 2, 3, 4)) == ArrayList<Int64>(0, 1, 2, 4, 3),
            sortArray(ArrayList<Int64>()) == ArrayList<Int64>(),
            sortArray(ArrayList<Int64>(2, 5, 77, 4, 5, 3, 5, 7, 2, 3, 4)) == ArrayList<Int64>(2, 2, 4, 4, 3, 3, 5, 5, 5, 7, 77),
            sortArray(ArrayList<Int64>(3, 6, 44, 12, 32, 5)) == ArrayList<Int64>(32, 3, 5, 6, 12, 44),
            sortArray(ArrayList<Int64>(2, 4, 8, 16, 32)) == ArrayList<Int64>(2, 4, 8, 16, 32),
            sortArray(ArrayList<Int64>(2, 4, 8, 16, 32)) == ArrayList<Int64>(2, 4, 8, 16, 32)
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