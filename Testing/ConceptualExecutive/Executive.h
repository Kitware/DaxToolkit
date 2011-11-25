#ifndef EXECUTIVE_H
#define EXECUTIVE_H


namespace dax { namespace exec
{
//workletHandle and worklet HandleT

class workletHandle
{
public:
  virtual ~workletHandle(){};
  virtual std::string name()const=0;
};

template<typename Worklet>
class workletHandleT : public workletHandle
{
private:
  Worklet Worklet_;

public:
  std::string name() const{ return Worklet_.name(); }

};

} }

class Executive
{
private:
  typedef std::vector< dax::exec::workletHandle* >  TaskVector;
  typedef std::vector< dax::exec::workletHandle* >::iterator  TaskIterator;

  TaskVector Tasks;
public:

  void run()
  {
    std::cout << "dumping the worklets to run" << std::endl;
    for(TaskIterator it=Tasks.begin();
        it!=Tasks.end();
        ++it)
      {
      std::cout << (*it)->name() << std::endl;
      }
  }

  template<typename Worklet>
  void add()
  {
    Tasks.push_back( new dax::exec::workletHandleT<Worklet>());
  }

};

#endif // EXECUTIVE_H
