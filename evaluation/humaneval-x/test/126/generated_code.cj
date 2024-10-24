main() {
    try{
    let correct = ArrayList<Bool>(
            isSorted(ArrayList<Int64>([5])) == true,
            isSorted(ArrayList<Int64>(1, 2, 3, 4, 5)) == true,
            isSorted(ArrayList<Int64>(1, 3, 2, 4, 5)) == false,
            isSorted(ArrayList<Int64>(1, 2, 3, 4, 5, 6)) == true,
            isSorted(ArrayList<Int64>(1, 2, 3, 4, 5, 6, 7)) == true,
            isSorted(ArrayList<Int64>(1, 3, 2, 4, 5, 6, 7)) == false,
            isSorted(ArrayList<Int64>()) == true,
            isSorted(ArrayList<Int64>([1])) == true,
            isSorted(ArrayList<Int64>(3, 2, 1)) == false,
            isSorted(ArrayList<Int64>(1, 2, 2, 2, 3, 4)) == false,
            isSorted(ArrayList<Int64>(1, 2, 3, 3, 3, 4)) == false,
            isSorted(ArrayList<Int64>(1, 2, 2, 3, 3, 4)) == true,
            isSorted(ArrayList<Int64>(1, 2, 3, 4)) == true
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