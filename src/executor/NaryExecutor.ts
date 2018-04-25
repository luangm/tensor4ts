import NaryOp from "../op/nary/NaryOp";

export class NaryExecutor {

  public exec(op: NaryOp): void {
    // switch (op.result.rank) {
    //   default:
    //     this.exec9General(op);
    //     break;
    // }
  }

  // For now, performance will be bad.
  // private exec9General(op: NaryOp): void {
  //   let result = op.result;
  //   for (let i = 0; i < result.length; i++) {
  //     let values = op.list.map(item => item.get(i));
  //     result.set(i, op.body(values));
  //   }
  // }

}

export default new NaryExecutor();