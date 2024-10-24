main() {
    try{
    let correct = ArrayList<Bool>(
            sortArray(ArrayList<Int64>())==ArrayList<Int64>(),
            sortArray(ArrayList<Int64>([5]))==ArrayList<Int64>([5]),
            sortArray(ArrayList<Int64>(2, 4, 3, 0, 1, 5))==ArrayList<Int64>(0, 1, 2, 3, 4, 5),
            sortArray(ArrayList<Int64>(2, 4, 3, 0, 1, 5, 6))==ArrayList<Int64>(6, 5, 4, 3, 2, 1, 0),
            sortArray(ArrayList<Int64>(2, 1))==ArrayList<Int64>(1, 2),
            sortArray(ArrayList<Int64>(15, 42, 87, 32 ,11, 0))==ArrayList<Int64>(0, 11, 15, 32, 42, 87),
            sortArray(ArrayList<Int64>(21, 14, 23, 11))==ArrayList<Int64>(23, 21, 14, 11)
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