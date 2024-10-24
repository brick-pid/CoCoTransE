main() {
    try{
    let correct = ArrayList<Bool>(
            add(ArrayList<Int64>(4, 88)) == 88,
            add(ArrayList<Int64>(4, 5, 6, 7, 2, 122)) == 122,
            add(ArrayList<Int64>(4, 0, 6, 7)) == 0,
            add(ArrayList<Int64>(4, 4, 6, 8)) == 12
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