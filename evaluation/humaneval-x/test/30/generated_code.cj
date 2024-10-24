main() {
    try{
    let correct = ArrayList<Bool>(
            getPositive(ArrayList<Int64>(-1, -2, 4, 5, 6)) == (ArrayList<Int64>(4, 5, 6)),
            getPositive(ArrayList<Int64>(5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10)) == (ArrayList<Int64>(5, 3, 2, 3, 3, 9, 123, 1)),
            getPositive(ArrayList<Int64>(-1, -2)) == (ArrayList<Int64>()),
            getPositive(ArrayList<Int64>()) == (ArrayList<Int64>())
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