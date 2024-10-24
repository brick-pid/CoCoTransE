main() {
    try{
    let correct = ArrayList<Bool>(
            rollingMax(ArrayList<Int64>()) == (ArrayList<Int64>()),
            rollingMax(ArrayList<Int64>(1, 2, 3, 4)) == (ArrayList<Int64>(1, 2, 3, 4)),
            rollingMax(ArrayList<Int64>(4, 3, 2, 1)) == (ArrayList<Int64>(4, 4, 4, 4)),
            rollingMax(ArrayList<Int64>(3, 2, 3, 100, 3)) == (ArrayList<Int64>(3, 3, 3, 100, 100))
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