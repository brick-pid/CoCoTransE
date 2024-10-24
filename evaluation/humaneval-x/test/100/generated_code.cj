main() {
    try{
    let correct = ArrayList<Bool>(
            makeAPile(3) == ArrayList<Int64>(3, 5, 7),
            makeAPile(4) == ArrayList<Int64>(4, 6, 8, 10),
            makeAPile(5) == ArrayList<Int64>(5, 7, 9, 11, 13),
            makeAPile(6) == ArrayList<Int64>(6, 8, 10, 12, 14, 16),
            makeAPile(8) == ArrayList<Int64>(8, 10, 12, 14, 16, 18, 20, 22)
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