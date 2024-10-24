main() {
    try{
    let correct = ArrayList<Bool>(
            fizzBuzz(50) == 0,
            fizzBuzz(78) == 2,
            fizzBuzz(79) == 3,
            fizzBuzz(100) == 3,
            fizzBuzz(200) == 6,
            fizzBuzz(4000) == 192,
            fizzBuzz(10000) == 639,
            fizzBuzz(100000) == 8026
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