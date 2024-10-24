main() {
    try{
    let correct = ArrayList<Bool>(
            isNested("[[]]" ),
            !isNested("[]]]]]]][[[[[]" ),
            !isNested("[][]" ),
            !isNested("[]" ),
            isNested("[[[[]]]]" ),
            !isNested("[]]]]]]]]]]" ),
            isNested("[][][[]]" ),
            !isNested("[[]" ),
            !isNested("[]]" ),
            isNested("[[]][[" ),
            isNested("[[][]]" ),
            !isNested("" ),
            !isNested("[[[[[[[[" ),
            !isNested("]]]]]]]]" )
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