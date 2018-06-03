// mut x: &String  vs   x: &mut String
// mut x: &String  -  x is mutable, x can be reassigned
// x: &mut String  -  content of x is mutable, x can be modified
fn f(mut s1: String, s2: &mut String) -> String {
  s1.push_str("[mut s1]");
  s2.push_str("[mut s2]");
  let mut s3 = String::from(s1.as_str());
  s3.push_str(s2.as_str());
  s3 // &s3 will fail. s3 will be out of scope before moving ownership out.
}

fn main() {
  let hello = String::from("Hello, V!");
  println!("{}", hello);

  let mut count = 0;  // Mutable need to use `mut`
  println!("Count is {}", count);
  count += 1;
  println!("Count is {}", count);
  let count = "NaN";  // Shadowing. Redeclare the variable.
  println!("Count is {}", count);


  let s1 = String::from("s1");
  let mut s2 = String::from("s2");
  let s3 = f(s1, &mut s2);
  println!("s3 is {}", s3);

  // s1.push_str("!"); will fail. s1 ownership is moved
  s2.push_str("!");  // fine. s2 ownership not moved.


  let s4 = s3; // move s3 ownership to s4. s3 not usable now.
  // println!("s3 is {}", s3); will fail
  println!("s4 is {} (should equals s3)", s4);
}
