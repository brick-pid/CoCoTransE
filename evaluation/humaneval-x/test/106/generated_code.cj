main() {
    try{
    let correct = ArrayList<Bool>(
            f(5)==ArrayList<Int64>(1, 2, 6, 24, 15),
            f(7)==ArrayList<Int64>(1, 2, 6, 24, 15, 720, 28),
            f(1)==ArrayList<Int64>([1]),
            f(3)==ArrayList<Int64>(1, 2, 6)
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