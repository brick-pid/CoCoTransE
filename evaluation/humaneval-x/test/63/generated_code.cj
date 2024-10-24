main() {
    try{
    let correct = ArrayList<Bool>(
            fibfib(2) == 1,
            fibfib(1) == 0,
            fibfib(5) == 4,
            fibfib(8) == 24,
            fibfib(10) == 81,
            fibfib(12) == 274,
            fibfib(14) == 927
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