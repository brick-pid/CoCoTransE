main() {
    try{
    let correct = ArrayList<Bool>(
            maxElement(ArrayList<Int64>(1, 2, 3)) == 3,
            maxElement(ArrayList<Int64>(5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10)) == 124
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