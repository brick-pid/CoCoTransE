main() {
    try{
    let correct = ArrayList<Bool>(
            belowThreshold(ArrayList<Int64>(1, 2, 4, 10), 100),
            !belowThreshold(ArrayList<Int64>(1, 20, 4, 10), 5),
            belowThreshold(ArrayList<Int64>(1, 20, 4, 10), 21),
            belowThreshold(ArrayList<Int64>(1, 20, 4, 10), 22),
            belowThreshold(ArrayList<Int64>(1, 8, 4, 10), 11),
            !belowThreshold(ArrayList<Int64>(1, 8, 4, 10), 10)
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