main() {
    try{
    let correct = ArrayList<Bool>(
            maxFill(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 0, 1, 0), ArrayList<Int64>(0, 1, 0, 0), ArrayList<Int64>(1, 1, 1, 1)), 1) == 6,
            maxFill(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 0, 1, 1), ArrayList<Int64>(0, 0, 0, 0), ArrayList<Int64>(1, 1, 1, 1), ArrayList<Int64>(0, 1, 1, 1)), 2) == 5,
            maxFill(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 0, 0), ArrayList<Int64>(0, 0, 0)), 5) == 0,
            maxFill(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(1, 1, 1, 1), ArrayList<Int64>(1, 1, 1, 1)), 2) == 4,
            maxFill(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(1, 1, 1, 1), ArrayList<Int64>(1, 1, 1, 1)), 9) == 2
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