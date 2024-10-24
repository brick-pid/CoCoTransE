main() {
    try{
    let correct = ArrayList<Bool>(
            nextSmallest(ArrayList<Int64>(1, 2, 3, 4, 5)).getOrThrow() == 2,
            nextSmallest(ArrayList<Int64>(5, 1, 4, 3, 2)).getOrThrow() == 2,
            nextSmallest(ArrayList<Int64>())==None,
            nextSmallest(ArrayList<Int64>(1, 1))==None,
            nextSmallest(ArrayList<Int64>(1, 1, 1, 1, 0)).getOrThrow() == 1,
            nextSmallest(ArrayList<Int64>(1, Int64(math.pow(0.0, 0.0))))==None,
            nextSmallest(ArrayList<Int64>(-35, 34, 12, -45)).getOrThrow() == -35
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