(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-0a0de00d"],{"901a":function(t,e,a){},f43a:function(t,e,a){"use strict";a.d(e,"n",(function(){return s})),a.d(e,"h",(function(){return n})),a.d(e,"i",(function(){return o})),a.d(e,"j",(function(){return r})),a.d(e,"o",(function(){return c})),a.d(e,"g",(function(){return d})),a.d(e,"k",(function(){return p})),a.d(e,"f",(function(){return u})),a.d(e,"c",(function(){return m})),a.d(e,"e",(function(){return f})),a.d(e,"m",(function(){return g})),a.d(e,"l",(function(){return v})),a.d(e,"b",(function(){return h})),a.d(e,"d",(function(){return b})),a.d(e,"a",(function(){return S}));a("99af");var i=a("b775"),l="http://127.0.0.1:3333/";function s(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/starttrain"),method:"post",data:t})}function n(){return Object(i["a"])({url:"".concat(l,"SilasGUI/trainPage/getTrainResult"),method:"get"})}function o(){return Object(i["a"])({url:"".concat(l,"SilasGUI/trainPage/getTreeData"),method:"get"})}function r(){return Object(i["a"])({url:"".concat(l,"SilasGUI/testPage/getValResult"),method:"get"})}function c(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/startvalidation"),method:"post",data:t})}function d(){return Object(i["a"])({url:"".concat(l,"SilasGUI/testPage/getTreeData"),method:"get"})}function p(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/predictPage/prediction"),method:"post",data:t})}function u(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/predictPage/getRes?type=").concat(t.type),method:"get"})}function m(){return Object(i["a"])({url:"".concat(l,"SilasGUI/gridSearchPage/gengridSearch"),method:"get"})}function f(){return Object(i["a"])({url:"".concat(l,"SilasGUI/gridSearchPage/getGridSearchRes"),method:"get"})}function g(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/gridSearchPage/gridSearch"),method:"post",data:t})}function v(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/extensionPage/explanation"),method:"post",data:t})}function h(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/featurePage/featureImportance"),method:"post",data:t})}function b(){return Object(i["a"])({url:"".concat(l,"SilasGUI/getStastisticData"),method:"get"})}function S(t){return Object(i["a"])({url:"".concat(l,"SilasGUI/deleteFile"),method:"get",params:t})}},f64e:function(t,e,a){"use strict";a("901a")},fb2c:function(t,e,a){"use strict";a.r(e);var i=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticStyle:{margin:"20px"}},[a("aside",[a("i",{staticClass:"el-icon-info"}),t._v(" "+t._s(t.$t("featurepage.intro"))+" "),a("a",{staticStyle:{"text-decoration":"underline"},attrs:{href:"https://github.com/Yunkun-Zhang/MUC-Silas",target:"_blank"}},[t._v(t._s(t.$t("global.sourcecode")))])]),a("el-button",{staticClass:"tutorial_btn",attrs:{type:"warning",icon:"el-icon-guide"},on:{click:function(e){return e.preventDefault(),e.stopPropagation(),t.guide(e)}}},[t._v(t._s(t.$t("global.guide")))]),a("div",{staticClass:"feature-importance-page"},[a("div",{staticClass:"feature-importance",staticStyle:{width:"48%"}},[a("el-card",{attrs:{id:"featureImportanceSetting"}},[a("h3",[t._v(t._s(t.$t("global.parameterSetting")))]),a("el-divider"),a("el-form",{staticClass:"model-explanaton-form",staticStyle:{"margin-bottom":"50px"},attrs:{"label-width":"150px"}},[a("div",{staticStyle:{display:"flex","justify-content":"space-around"},attrs:{id:"featureImportanceUpload"}},[a("el-form-item",{attrs:{label:"Model Path:"}},[a("el-select",{model:{value:t.modelPath,callback:function(e){t.modelPath=e},expression:"modelPath"}},[a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: silas-temp/results/model",placement:"right-start"}},[a("el-option",{attrs:{value:"silas-temp/results/model",label:"Trained model"}})],1),a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: silas-temp/validation-results/model",placement:"right-start"}},[a("el-option",{attrs:{value:"silas-temp/validation-results/model",label:"Validated model"}})],1),a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: OptExplain/optexplain-example/models",placement:"right-start"}},[a("el-option",{attrs:{value:"OptExplain/optexplain-example/models",label:"Example model"}})],1)],1)],1),a("el-form-item",{attrs:{label:t.$t("global.testingFile")+": "}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{"text-align":"left"},attrs:{data:{fileName:"featureImportance_test.csv"},action:"http://127.0.0.1:3333/SilasGUI/uploadFile","file-list":t.fileList,limit:1,accept:".csv","on-success":t.handleSuccess,"on-remove":t.handleRemove}},[t.canFeaClick?a("el-button",{attrs:{slot:"tip",size:"small",type:"primary",disabled:""},slot:"tip"},[a("i",{staticClass:"el-icon-upload"}),t._v(" "+t._s(t.$t("global.testingFile")))]):a("el-button",{attrs:{size:"small",type:"primary"}},[a("i",{staticClass:"el-icon-upload"}),t._v(" "+t._s(t.$t("global.testingFile")))])],1)],1)],1),a("el-button",{staticClass:"feature-im-btn",staticStyle:{float:"right"},attrs:{id:"featureImportance"},on:{click:function(e){return t.explaination("feature")}}},[t._v(t._s(t.$t("featurepage.featureimportancetitle"))+" ")])],1)],1),a("div",{directives:[{name:"loading",rawName:"v-loading",value:t.showLoading,expression:"showLoading"}],staticStyle:{"margin-top":"30px","text-align":"center"},attrs:{id:"featureImportanceRes"}},[a("h3",[t._v(t._s(t.$t("global.result")))]),a("el-divider"),a("div",{staticStyle:{height:"calc(100vh - 360px)",overflow:"auto"}},[a("el-table",{staticClass:"sub-table",attrs:{data:t.featureImportanceList}},[a("el-table-column",{attrs:{type:"expand"},scopedSlots:t._u([{key:"default",fn:function(t){return[a("el-table",{attrs:{data:t.row.details}},[a("el-table-column",{attrs:{label:"Feature Name",prop:"name"}}),a("el-table-column",{attrs:{label:"Importantce Value",prop:"value"}})],1)]}}])}),a("el-table-column",{attrs:{label:"Class Name",prop:"class"}}),a("el-table-column",{attrs:{label:"Sub Feature Number",prop:"sublenght"}})],1)],1)],1)],1),a("el-divider",{staticStyle:{"margin-left":"30px",height:"calc(100vh - 200px)"},attrs:{direction:"vertical"}}),a("div",{staticClass:"adversary-sample",staticStyle:{width:"48%","margin-left":"20px"}},[a("el-card",{attrs:{id:"advSampleSetting"}},[a("h3",[t._v(t._s(t.$t("global.parameterSetting")))]),a("el-divider"),a("el-form",{staticClass:"model-explanaton-form",staticStyle:{"margin-bottom":"50px"},attrs:{"label-width":"150px"}},[a("div",{staticStyle:{display:"flex","justify-content":"space-around"},attrs:{id:"advSampleUpload"}},[a("el-form-item",{attrs:{label:"Model Path:"}},[a("el-select",{model:{value:t.modelPath_for_adv,callback:function(e){t.modelPath_for_adv=e},expression:"modelPath_for_adv"}},[a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: silas-temp/results/model",placement:"right-start"}},[a("el-option",{attrs:{value:"silas-temp/results/model",label:"Trained model"}})],1),a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: silas-temp/validation-results/model",placement:"right-start"}},[a("el-option",{attrs:{value:"silas-temp/validation-results/model",label:"Validated model"}})],1),a("el-tooltip",{staticClass:"item",attrs:{effect:"dark",content:"path: OptExplain/optexplain-example/models",placement:"right-start"}},[a("el-option",{attrs:{value:"OptExplain/optexplain-example/models",label:"Example model"}})],1)],1)],1),a("el-form-item",{attrs:{label:t.$t("global.testingFile")+": "}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{"text-align":"left"},attrs:{data:{fileName:"adversial_test.csv"},action:"http://127.0.0.1:3333/SilasGUI/uploadFile","file-list":t.fileList,limit:1,accept:".csv","on-success":t.handleAdvSuccess,"on-remove":t.handleAdvRemove}},[t.canAdvClick?a("el-button",{attrs:{slot:"tip",size:"small",type:"primary",disabled:""},slot:"tip"},[a("i",{staticClass:"el-icon-upload"}),t._v(" "+t._s(t.$t("global.testingFile")))]):a("el-button",{attrs:{size:"small",type:"primary"}},[a("i",{staticClass:"el-icon-upload"}),t._v(" "+t._s(t.$t("global.testingFile")))])],1)],1)],1),a("el-button",{staticClass:"feature-im-btn",staticStyle:{float:"right"},attrs:{id:"advSample"},on:{click:function(e){return t.explaination("adversarial")}}},[t._v(t._s(t.$t("advpage.advtitle"))+" ")])],1)],1),a("div",{directives:[{name:"loading",rawName:"v-loading",value:t.showAdvLoading,expression:"showAdvLoading"}],staticStyle:{"margin-top":"30px","text-align":"center"},attrs:{id:"advSampleRes"}},[a("h3",[t._v(t._s(t.$t("global.result")))]),a("el-divider"),a("div",{staticStyle:{display:"flex","margin-top":"20px","justify-content":"space-around"}},[a("el-card",{staticClass:"res-card"},[a("span",{staticStyle:{"font-size":"12px"}},[t._v("Original class:")]),a("br"),a("span",{staticStyle:{"line-height":"50px"},domProps:{innerHTML:t._s(t.adversarialObj.orgClass)}})]),a("el-card",{staticClass:"res-card",staticStyle:{margin:"0 10px"}},[a("span",{staticStyle:{"font-size":"12px"}},[t._v("Adv sample class:")]),a("br"),a("span",{staticStyle:{"line-height":"50px"},domProps:{innerHTML:t._s(t.adversarialObj.advSamClass)}})]),a("el-card",{staticClass:"res-card"},[a("span",{staticStyle:{"font-size":"12px"}},[t._v("Distance")]),a("br"),a("span",{staticStyle:{"line-height":"50px"},domProps:{innerHTML:t._s(t.adversarialObj.Distance)}})])],1)],1)],1)],1)],1)},l=[],s=a("c24c"),n=a.n(s);a("01d77");function o(t){return[{element:"#featureImportanceSetting",popover:{title:t.t("featurepage.settingtitle"),description:t.t("featurepage.settingdesc"),position:"bottom"}},{element:"#featureImportanceUpload",popover:{title:t.t("featurepage.uploadtitle"),description:t.t("featurepage.uploaddesc"),position:"bottom"}},{element:"#featureImportance",popover:{title:t.t("featurepage.featureimportancetitle"),description:t.t("featurepage.featureimportancedesc"),position:"bottom"}},{element:"#featureImportanceRes",popover:{title:t.t("featurepage.featureimportancerestitle"),description:t.t("featurepage.featureimportancerestitle"),position:"top"}},{element:"#advSampleSetting",popover:{title:t.t("advpage.settingtitle"),description:t.t("advpage.settingdesc"),position:"bottom"}},{element:"#advSampleUpload",popover:{title:t.t("advpage.uploadtitle"),description:t.t("advpage.uploaddesc"),position:"bottom"}},{element:"#advSample",popover:{title:t.t("advpage.advtitle"),description:t.t("advpage.advdesc"),position:"bottom"}},{element:"#advSampleRes",popover:{title:t.t("advpage.advrestitle"),description:t.t("advpage.advresdesc"),position:"bottom"}}]}var r=a("f43a"),c={data:function(){return{modelPath:"OptExplain/optexplain-example/models",modelPath_for_adv:"OptExplain/optexplain-example/models",testFilePath:"OptExplain/optexplain-example/test.csv",PSOIterationsNum:20,PSOParticlesNum:20,accWeight:.5,conjunction:"yes",explanationData:[],showPSO:!1,showLoading:!1,showAdvLoading:!1,fileList:[],performance:{Coverage:"-","EX AUC":"-","EX accuracy":"-",Overlap:"-",Performance:"-","RF AUC":"-","RF accuracy":"-","Sample size":"-"},featureImportanceList:[],adversarialObj:{Distance:"0.0",advSamClass:0,opt_sample:[6,147.64441628529508,72,35,0,26.210932884657822,.6387911514805615,49.92216384827193],orgClass:"1",x:[6,148,72,35,0,33.6,.627,50]},canAdvClick:!1,canFeaClick:!1,posStr:"1   10  [0.681 0.103 0.347 1.833]\t 0.81770833  4   fitness: 0.89736548\n2   6   [0.292 0.124 0.267 1.   ]\t 0.81770833  4   fitness: 0.89736548\n3   4   [0.674 0.133 0.344 1.768]\t 0.81770833  4   fitness: 0.89736548\n4   4   [0.758 0.104 0.344 1.23 ]\t 0.81770833  4   fitness: 0.89736548\n5   11  [0.608 0.106 0.422 1.   ]\t 0.83203125  4   fitness: 0.90452694\n6   11  [0.604 0.099 0.432 1.   ]\t 0.83203125  4   fitness: 0.90452694\n7   0   [0.589 0.106 0.497 1.   ]\t 0.83203125  4   fitness: 0.90452694\n8   0   [0.536 0.105 0.55  1.   ]\t 0.81770833  4   fitness: 0.89736548\n9   2   [0.64  0.098 0.417 1.   ]\t 0.81770833  4   fitness: 0.89736548\n10  5   [0.617 0.094 0.37  1.   ]\t 0.85807292  4   fitness: 0.91754777\n11  0   [0.607 0.089 0.379 1.   ]\t 0.85807292  4   fitness: 0.91754777\n12  5   [0.59  0.099 0.395 1.   ]\t 0.81770833  4   fitness: 0.89736548\n13  10  [0.646 0.091 0.335 1.   ]\t 0.85807292  4   fitness: 0.91754777\n14  5   [0.64  0.089 0.35  1.   ]\t 0.85807292  4   fitness: 0.91754777\n15  0   [0.615 0.093 0.375 1.   ]\t 0.85807292  4   fitness: 0.91754777\n16  1   [0.611 0.093 0.378 1.   ]\t 0.85807292  4   fitness: 0.91754777\n17  11  [0.622 0.091 0.346 1.   ]\t 0.85807292  4   fitness: 0.91754777\n18  0   [0.611 0.092 0.379 1.   ]\t 0.85807292  4   fitness: 0.91754777\n19  0   [0.616 0.094 0.373 1.   ]\t 0.85807292  4   fitness: 0.91754777\n20  0   [0.613 0.092 0.373 1.   ]\t 0.85807292  4   fitness: 0.91754777\nOptimal parameters: [0.617 0.094 0.37  1.   ]\n20  0   [0.613 0.092 0.373 1.   ]\t 0.85807292  4   fitness: 0.91754777"}},mounted:function(){this.driver=new n.a},methods:{guide:function(){var t=o(this.$i18n);this.driver.defineSteps(t),this.driver.start()},handleSuccess:function(){this.canFeaClick=!0},handleRemove:function(){this.canFeaClick=!1},handleAdvSuccess:function(){this.canAdvClick=!0},handleAdvRemove:function(){this.canAdvClick=!1},explaination:function(t){console.log("type",t);var e=this,a=e.modelPath;"adversarial"===t?(a=e.modelPath_for_adv,this.showAdvLoading=!0):this.showLoading=!0,Object(r["b"])({type:t,model_path:a,test_file_path:e.testFilePath}).then((function(a){"feature"==t?(console.log("resss",a),e.featureImportanceList=a.featureImportance):e.adversarialObj=a.adversarialObj,e.showLoading=!1,e.showAdvLoading=!1}))}}},d=c,p=(a("f64e"),a("2877")),u=Object(p["a"])(d,i,l,!1,null,null,null);e["default"]=u.exports}}]);