import TransformOp from "./TransformOp";

export default class LogOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.log(a);
  }

}