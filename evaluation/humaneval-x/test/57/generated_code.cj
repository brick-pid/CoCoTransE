main() {
    try{
    let correct = ArrayList<Bool>(
            monotonic(ArrayList<Int64>(1, 2, 4, 10)),
            monotonic(ArrayList<Int64>(1, 2, 4, 20)),
            !monotonic(ArrayList<Int64>(1, 20, 4, 10)),
            monotonic(ArrayList<Int64>(4, 1, 0, -10)),
            monotonic(ArrayList<Int64>(4, 1, 1, 0)),
            !monotonic(ArrayList<Int64>(1, 2, 3, 2, 5, 60)),
            monotonic(ArrayList<Int64>(1, 2, 3, 4, 5, 60)),
            monotonic(ArrayList<Int64>(9, 9, 9, 9))
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