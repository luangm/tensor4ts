import TransformOp from "./TransformOp";

export default class TanhGradOp extends TransformOp {

  body(a: number, b?: number): number {
    let sech = 1 / Math.cosh(a);
    return sech * sech;
  }

}