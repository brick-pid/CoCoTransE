main() {
    try{
    let correct = ArrayList<Bool>(
            moveOneBall(ArrayList<Int64>(3, 4, 5, 1, 2)) == true,
            moveOneBall(ArrayList<Int64>(3, 5, 10, 1, 2)) == true,
            moveOneBall(ArrayList<Int64>(4, 3, 1, 2)) == false,
            moveOneBall(ArrayList<Int64>(3, 5, 4, 1, 2)) == false,
            moveOneBall(ArrayList<Int64>()) == true
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