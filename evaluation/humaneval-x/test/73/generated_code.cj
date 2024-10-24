main() {
    try{
    let correct = ArrayList<Bool>(
            smallestChange(ArrayList<Int64>(1, 2, 3, 5, 4, 7, 9, 6)) == 4,
            smallestChange(ArrayList<Int64>(1, 2, 3, 4, 3, 2, 2)) == 1,
            smallestChange(ArrayList<Int64>(1, 4, 2)) == 1,
            smallestChange(ArrayList<Int64>(1, 4, 4, 2)) == 1,
            smallestChange(ArrayList<Int64>(1, 2, 3, 2, 1)) == 0,
            smallestChange(ArrayList<Int64>(3, 1, 1, 3)) == 0,
            smallestChange(ArrayList<Int64>([1])) == 0,
            smallestChange(ArrayList<Int64>(0, 1)) == 1
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