main() {
    try{
    let correct = ArrayList<Bool>(
            intersperse(ArrayList<Int64>(), 7) == (ArrayList<Int64>()),
            intersperse(ArrayList<Int64>(5, 6, 3, 2), 8) == (ArrayList<Int64>(5, 8, 6, 8, 3, 8, 2)),
            intersperse(ArrayList<Int64>(2, 2, 2), 2) == (ArrayList<Int64>(2, 2, 2, 2, 2))
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