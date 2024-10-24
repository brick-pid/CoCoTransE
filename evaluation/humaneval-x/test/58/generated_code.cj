main() {
    try{
    let correct = ArrayList<Bool>(
            common(ArrayList<Int64>(1, 4, 3, 34, 653, 2, 5), ArrayList<Int64>(5, 7, 1, 5, 9, 653, 121)) == (ArrayList<Int64>(1, 5, 653)),
            common(ArrayList<Int64>(5, 3, 2, 8), ArrayList<Int64>(3, 2)) == (ArrayList<Int64>(2, 3)),
            common(ArrayList<Int64>(4, 3, 2, 8), ArrayList<Int64>(3, 2, 4)) == (ArrayList<Int64>(2, 3, 4)),
            common(ArrayList<Int64>(4, 3, 2, 8), ArrayList<Int64>()) == (ArrayList<Int64>())
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