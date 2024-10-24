main() {
    try{
    let correct = ArrayList<Bool>(
            incrList(ArrayList<Int64>()) == (ArrayList<Int64>()),
            incrList(ArrayList<Int64>(3, 2, 1)) == (ArrayList<Int64>(4, 3, 2)),
            incrList(ArrayList<Int64>(5, 2, 5, 2, 3, 3, 9, 0, 123)) == (ArrayList<Int64>(6, 3, 6, 3, 4, 4, 10, 1, 124))
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