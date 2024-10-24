main() {
    try{
    let correct = ArrayList<Bool>(
            removeDuplicates(ArrayList<Int64>()) == (ArrayList<Int64>()),
            removeDuplicates(ArrayList<Int64>(1, 2, 3, 4)) == (ArrayList<Int64>(1, 2, 3, 4)),
            removeDuplicates(ArrayList<Int64>(1, 2, 3, 2, 4, 3, 5)) == (ArrayList<Int64>(1, 4, 5))
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