main() {
    try{
    let correct = ArrayList<Bool>(
            sumProduct(ArrayList<Int64>()) == (ArrayList<Int64>(0, 1)),
            sumProduct(ArrayList<Int64>(1, 1, 1)) == (ArrayList<Int64>(3, 1)),
            sumProduct(ArrayList<Int64>(100, 0)) == (ArrayList<Int64>(100, 0)),
            sumProduct(ArrayList<Int64>(3, 5, 7)) == (ArrayList<Int64>(3 + 5 + 7, 3 * 5 * 7)),
            sumProduct(ArrayList<Int64>([10])) == (ArrayList<Int64>(10, 10))
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