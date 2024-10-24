main() {
    try{
    let correct = ArrayList<Bool>(
            reverseDelete("abcde", "ae" ).equals(Arrays.asList("bcd", false)),
            reverseDelete("abcdef", "b" ).equals(Arrays.asList("acdef", false)),
            reverseDelete("abcdedcba", "ab" ).equals(Arrays.asList("cdedc", true)),
            reverseDelete("dwik", "w" ).equals(Arrays.asList("dik", false)),
            reverseDelete("a", "a" ).equals(Arrays.asList("", true)),
            reverseDelete("abcdedcba", "" ).equals(Arrays.asList("abcdedcba", true)),
            reverseDelete("abcdedcba", "v" ).equals(Arrays.asList("abcdedcba", true)),
            reverseDelete("vabba", "v" ).equals(Arrays.asList("abba", true)),
            reverseDelete("mamma", "mia" ).equals(Arrays.asList("", true))
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