main() {
    try{
    let correct = ArrayList<Bool>(
            canArrange(ArrayList<Int64>(1, 2, 4, 3, 5)) == 3,
            canArrange(ArrayList<Int64>(1, 2, 4, 5)) == -1,
            canArrange(ArrayList<Int64>(1, 4, 2, 5, 6, 7, 8, 9, 10)) == 2,
            canArrange(ArrayList<Int64>(4, 8, 5, 7, 3)) == 4,
            canArrange(ArrayList<Int64>()) == -1
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