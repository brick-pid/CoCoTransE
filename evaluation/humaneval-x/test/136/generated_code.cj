main() {
    try{
    let correct = ArrayList<Bool>(
            largestSmallestintegers(ArrayList<Int64>(2, 4, 1, 3, 5, 7))==ArrayList<Option<Int64>>(None, Some(1)),
            largestSmallestintegers(ArrayList<Int64>(2, 4, 1, 3, 5, 7, 0))==ArrayList<Option<Int64>>(None, Some(1)),
            largestSmallestintegers(ArrayList<Int64>(1, 3, 2, 4, 5, 6, -2))==ArrayList<Option<Int64>>(Some(-2), Some(1)),
            largestSmallestintegers(ArrayList<Int64>(4, 5, 3, 6, 2, 7, -7))==ArrayList<Option<Int64>>(Some(-7), Some(2)),
            largestSmallestintegers(ArrayList<Int64>(7, 3, 8, 4, 9, 2, 5, -9))==ArrayList<Option<Int64>>(Some(-9), Some(2)),
            largestSmallestintegers(ArrayList<Int64>())==ArrayList<Option<Int64>>(None, None),
            largestSmallestintegers(ArrayList<Int64>([0]))==ArrayList<Option<Int64>>(None, None),
            largestSmallestintegers(ArrayList<Int64>(-1, -3, -5, -6))==ArrayList<Option<Int64>>(Some(-1), None),
            largestSmallestintegers(ArrayList<Int64>(-1, -3, -5, -6, 0))==ArrayList<Option<Int64>>(Some(-1), None),
            largestSmallestintegers(ArrayList<Int64>(-6, -4, -4, -3, 1))==ArrayList<Option<Int64>>(Some(-3), Some(1)),
            largestSmallestintegers(ArrayList<Int64>(-6, -4, -4, -3, -100, 1))==ArrayList<Option<Int64>>(Some(-3), Some(1))
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