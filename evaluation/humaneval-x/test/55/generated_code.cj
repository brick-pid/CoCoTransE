main() {
    try{
    let correct = ArrayList<Bool>(
            fib(10) == 55,
            fib(1) == 1,
            fib(8) == 21,
            fib(11) == 89,
            fib(12) == 144
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