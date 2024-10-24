main() {
    try{
    let correct = ArrayList<Bool>(
            evenOddCount(7)==ArrayList<Int64>(0, 1),
            evenOddCount(-78)==ArrayList<Int64>(1, 1),
            evenOddCount(3452)==ArrayList<Int64>(2, 2),
            evenOddCount(346211)==ArrayList<Int64>(3, 3),
            evenOddCount(-345821)==ArrayList<Int64>(3, 3),
            evenOddCount(-2)==ArrayList<Int64>(1, 0),
            evenOddCount(-45347)==ArrayList<Int64>(2, 3),
            evenOddCount(0)==ArrayList<Int64>(1, 0)
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