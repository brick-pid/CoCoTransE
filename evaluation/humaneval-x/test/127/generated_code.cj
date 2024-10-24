main() {
    try{
    let correct = ArrayList<Bool>(
            intersection(ArrayList<Int64>(1, 2), ArrayList<Int64>(2, 3)) == "NO",
            intersection(ArrayList<Int64>(-1, 1), ArrayList<Int64>(0, 4)) == "NO",
            intersection(ArrayList<Int64>(-3, -1), ArrayList<Int64>(-5, 5)) == "YES",
            intersection(ArrayList<Int64>(-2, 2), ArrayList<Int64>(-4, 0)) == "YES",
            intersection(ArrayList<Int64>(-11, 2), ArrayList<Int64>(-1, -1)) == "NO",
            intersection(ArrayList<Int64>(1, 2), ArrayList<Int64>(3, 5)) == "NO",
            intersection(ArrayList<Int64>(1, 2), ArrayList<Int64>(1, 2)) == "NO",
            intersection(ArrayList<Int64>(-2, -2), ArrayList<Int64>(-3, -2)) == "NO"
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