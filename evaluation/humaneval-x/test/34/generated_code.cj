main() {
    try{
    let correct = ArrayList<Bool>(
            unique(ArrayList<Int64>(5, 3, 5, 2, 3, 3, 9, 0, 123)) == (ArrayList<Int64>(0, 2, 3, 5, 9, 123))
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