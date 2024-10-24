main() {
    try{
    let correct = ArrayList<Bool>(
            eat(5, 6, 10)==ArrayList<Int64>(11, 4),
            eat(4, 8, 9)==ArrayList<Int64>(12, 1),
            eat(1, 10, 10)==ArrayList<Int64>(11, 0),
            eat(2, 11, 5)==ArrayList<Int64>(7, 0),
            eat(4, 5, 7)==ArrayList<Int64>(9, 2),
            eat(4, 5, 1)==ArrayList<Int64>(5, 0)
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