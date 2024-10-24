main() {
    try{
    let correct = ArrayList<Bool>(
            primeLength("Hello") == true,
            primeLength("abcdcba") == true,
            primeLength("kittens") == true,
            primeLength("orange") == false,
            primeLength("wow") == true,
            primeLength("world") == true,
            primeLength("MadaM") == true,
            primeLength("Wow") == true,
            primeLength("") == false,
            primeLength("HI") == true,
            primeLength("go") == true,
            primeLength("gogo") == false,
            primeLength("aaaaaaaaaaaaaaa") == false,
            primeLength("Madam") == true,
            primeLength("M") == false,
            primeLength("0") == false
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