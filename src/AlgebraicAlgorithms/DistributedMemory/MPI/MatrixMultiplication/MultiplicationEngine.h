/* Author: Edward Hutter */

#ifndef MULTIPLICATION_ENGINE_H_
#define MULTIPLICATION_ENGINE_H_

// We should use a host class and implement/define partial specialization classes
template<typename T, typename U>
class MultiplicationEngine
{
  // Lets prevent any instances of this class from being created.
public:
  MultiplicationEngine() = delete;
  MultiplicationEngine(const MultiplicationEngine& rhs) = delete;
  MultiplicationEngine(MultiplicationEnginer&& rhs) = delete;
  MultiplicationEngine<T,U> operator=(const MultiplicationEngine& rhs) = delete;
  MultiplicationEngine<T,U> operator=(MultiplicationEngine&& rhs) = delete;
  ~MultiplicationEngine() = delete;

  // Engine methods
  static void 

private:
}

#endif /* MULTIPLICATION_ENGINE_H_ */
